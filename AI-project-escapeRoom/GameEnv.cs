using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
using System.Diagnostics;
using System;
using System.IO;
using System.Threading;

class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 5000;
    private int currentStep;
    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
    }

    public double[] GetState()
    {
        // Example state representation
        return new double[]
        {
            game.player.Position.X / game.widthLevel,
            game.player.Position.Y / game.groundLevel,
            game.box.Position.X / game.widthLevel,
            game.box.Position.Y / game.groundLevel,
            game.IsPressed ? 1.0 : 0.0,
            game.IsOpen ? 1.0 : 0.0
        };
    }

    public (double[], int, bool) Step(int action)
    {
        currentStep++;
        int reward = 0;
        bool IsDone = false;

        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); game.player.IsGrounded = false; break; // Jump
            case 3: game.player.Grab(game.box); break; // Interact (e.g., pick up the box)
            case 4: game.player.DropHeldBox(); break; // Interact (e.g., drop the box)
        }
        ///////////////////////////////
        // Rewards for key objectives//
        ///////////////////////////////
        //console.Write("Reward: " + reward);
        // Small continuous rewards for correct behaviors
        if (game.IsMovingToward(game.box, game.lastPlayerPosition) && !game.IsPressed)
        {
            reward += 3; // Encourage moving toward the box
            ////console.Write("+" + 3);
        }
        else
        {
            reward -= 1; // Penalize moving away from the box
            //console.Write("-" + 1);
        }

        if (game.player.heldBox != null && game.IsMovingToward(game.button, game.lastPlayerPosition))
        {
            reward += 10; // Encourage moving toward button while holding the box
            //console.Write("+" + 10);
        }

        if (game.box.Intersects(game.button) && !game.previousBoxState)
        {
            reward += 50; // Reward for **placing** the box on the button
            game.IsPressed = true;
            game.previousBoxState = true; // Prevent continuous reward abuse
            //console.Write("+" + 100);
        }

        if (game.IsPressed && game.player.Intersects(game.door))
        {
            reward += 200; // Reward for completing the goal
            IsDone = true;
            //console.Write("+" + 200);
        }

        // Slight reward for progress toward the door **only** if button is pressed
        if (game.IsPressed && game.IsMovingToward(game.door, game.lastPlayerPosition))
        {
            reward += 7;
            //console.Write("+" + 7);
        }

        // **Penalties**
        if (!game.box.Intersects(game.button) && game.player.Intersects(game.door))
        {
            reward -= 20; // Lower penalty (was too harsh)
            //console.Write("-" + 20);
        }

        if (game.IsIdle())
        {
            reward -= 2; // Increased penalty for inactivity
        }

        // Out of bounds penalties
        if (IsOutOfBounds(game.player) || IsOutOfBounds(game.box))
        {
            reward -= 20;
            ResetPlayerAndBox();
            //console.Write("-" + 20);
        }

        //moveing away from goal panalty
        if (!game.IsMovingToward(game.door, game.lastPlayerPosition) && game.IsPressed
        || !game.IsMovingToward(game.box, game.lastPlayerPosition) && !game.IsPressed)
        {
            reward -= 15;
            //console.Write("-" + 15);
        }

        // Maximum steps penalty
        if (currentStep >= maxSteps)
        {
            reward -= 50; // Lowered penalty to allow learning
            ResetPlayerAndBox();
            IsDone = true;
            //console.Write("-" + 150);
        }


        //console.WriteLine("Total Reward: " + reward);
        Thread.Sleep(5);
        return (GetState(), reward, IsDone);
    }

    // Helper methods
    private bool IsOutOfBounds(GameObject obj)
    {
        return obj.Position.X < game.cameraPosition.X || obj.Position.X > game.cameraPosition.X + game.ScreenWidth ||
               obj.Position.Y < game.cameraPosition.Y || obj.Position.Y > game.cameraPosition.Y + game.ScreenHeight;
    }

    private void ResetPlayerAndBox()
    {
        ResetPlayerPosition();
        ResetBoxPosition();
        currentStep = 0;
    }

    private void ResetPlayerPosition()
    {
        game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);
    }

    private void ResetBoxPosition()
    {
        game.box.Position = new Vector2(200, game.groundLevel - game.box.Size.Y);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}