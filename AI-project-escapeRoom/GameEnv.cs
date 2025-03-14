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
using System.Collections.Generic;

class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 5000;
    private int currentStep;
    public List<int> lastPlayerMove;

    public double pick_the_box = 10;
    public double place_the_box_good = 10;
    public double finish_reward = 100;
    public double droping_box_bad = -1;
    public double culide_with_wall = -1;
    public double repeating_actions = -1;
    public double time_panalty = -1;
    public double max_steps_panalty = -10;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
        this.lastPlayerMove = new List<int>();
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

    public (double[], double, bool) Step(int action)
    {
        currentStep++;
        double reward = 0;
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

        lastPlayerMove.Add(action);
        ///////////////////////////////
        // Rewards for key objectives//
        ///////////////////////////////

        //pick the box
        if (game.player.Intersects(game.box) && game.player.heldBox == null)
        {
            reward += pick_the_box; // Reward for picking up the box
            Console.WriteLine("Picked up the box. Reward: " + pick_the_box);
        }

        //placing the box on the button
        if (game.box.Intersects(game.button) && game.player.heldBox == null)
        {
            reward += place_the_box_good; // Reward for placing the box on the button
            Console.WriteLine("Placed the box on the button. Reward: " + place_the_box_good);
        }

        //exiting the room finish goal
        if (game.IsPressed && IsOutOfBounds(game.player))
        {
            reward += finish_reward; // Reward for escaping the room
            IsDone = true;
            Console.WriteLine("Escaped the room. Reward: " + finish_reward);
        }

        ///////////////////////////////
        // Penalties for incorrect behaviors//
        ///////////////////////////////

        //droping the box for no resone
        if (game.player.heldBox == null && game.previousBoxState == true
        && !game.box.Intersects(game.button))
        {
            reward -= droping_box_bad; // Penalty for dropping the box for no reason
            Console.WriteLine("Dropped the box for no reason. Penalty: " + droping_box_bad);
        }

        //culiding with the walls (not the ground)
        if (!game.player.IsGrounded && game.player.Intersects(game.wall))
        {
            reward -= culide_with_wall; // Penalty for colliding with the walls
            Console.WriteLine("Collided with the walls. Penalty: " + culide_with_wall);
        }

        //repeating actions
        if (RepeatingActions(lastPlayerMove))
        {
            reward -= repeating_actions; // Penalty for repeating the same action
            Console.WriteLine("Repeated the same action. Penalty: " + repeating_actions);
        }

        //time penalty
        if (currentStep % 100 == 0)
        {
            reward -= time_panalty; // Penalty for taking too long
            Console.WriteLine("Taking too long. Penalty: " + time_panalty);
        }

        // Reset if out of bounds
        if (IsOutOfBounds(game.player) || IsOutOfBounds(game.box))
        {
            ResetPlayerAndBox();
            Console.WriteLine("Out of bounds. Resetting player and box.");
        }

        // Maximum steps penalty
        if (currentStep >= maxSteps)
        {
            reward -= max_steps_panalty; // Small penalty for exceeding maximum steps
            ResetPlayerAndBox();
            IsDone = true;
            currentStep = 0;
            Console.WriteLine("Exceeded maximum steps. Penalty: " + max_steps_panalty);
        }

        Thread.Sleep(1);
        return (GetState(), reward, IsDone);
    }

    public bool RepeatingActions(List<int> actions)
    {
        int length = 100;
        if (actions.Count < length * 2)
        {
            return false;
        }

        var lastActions = actions.GetRange(actions.Count - length, length);
        var previousActions = actions.GetRange(actions.Count - length * 2, length);

        return lastActions.SequenceEqual(previousActions);
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